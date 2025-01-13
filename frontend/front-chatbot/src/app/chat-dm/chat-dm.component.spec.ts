import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ChatDMComponent } from './chat-dm.component';

describe('ChatDMComponent', () => {
  let component: ChatDMComponent;
  let fixture: ComponentFixture<ChatDMComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ChatDMComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ChatDMComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
